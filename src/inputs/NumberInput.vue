<template>
  <div class="NumberInput">
    <div class="form-group">
      <label class="input-label">
        {{ label }}
        <input
          type="number"
          :min="min"
          :max="max"
          class="form-control"
          :class="{ 'is-invalid': validation && validation.$error }"
          :value="value"
          @input="$emit('input', $event.target.value)"
        />
      </label>

      <ValidationErrors
        v-if="validation && validation.$error"
        :validation="validation"
      />
    </div>
  </div>
</template>

<script lang="ts">
import { Validation } from 'vuelidate';
import { Component, Prop, Vue } from 'vue-property-decorator';
import ValidationErrors from './ValidationErrors.vue';

@Component({
  components: {
    ValidationErrors,
  },
})
export default class NumberInput extends Vue {
  @Prop({ required: true }) readonly label!: string;
  @Prop({ default: 0 }) readonly value!: number;
  @Prop({ default: 0 }) readonly min!: number;
  @Prop({ default: 100 }) readonly max!: number;
  @Prop({ required: true }) readonly validation!: Validation;
}
</script>
