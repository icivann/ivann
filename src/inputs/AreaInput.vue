<template>
  <div class="AreaInput">
    <div class="form-group">
      <label class="input-label">
        {{ label }}
        <textarea
          class="form-control"
          :class="{ 'is-invalid': validation && validation.$error }"
          @input="$emit('input', $event.target.value)"
          :value="value"
          :placeholder="placeholder"
          :rows="rows"
        >
        </textarea>
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
export default class AreaInput extends Vue {
  @Prop({ required: true }) readonly label!: string;
  @Prop({ required: true }) readonly placeholder!: string;
  @Prop({ required: true }) readonly rows!: number;
  @Prop({ required: true }) readonly validation!: Validation;
  @Prop({ default: '' }) readonly value!: string;
}
</script>
