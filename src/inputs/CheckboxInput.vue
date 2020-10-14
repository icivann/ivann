<template>
  <div class="CheckboxInput">
    <div class="form-group">
      <label>
        {{ label }}
        <input
          type="checkbox"
          class="form-check-input mr-1"
          :value="value"
          @input="$emit('input', $event.target.checked)"
        />
      </label>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component({})
export default class CheckboxInput extends Vue {
  @Prop({ required: true }) readonly label!: string;
  @Prop({ required: true }) readonly value!: string;
}
</script>
